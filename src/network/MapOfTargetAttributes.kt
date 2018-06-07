package network

data class MapOfTargetAttributes (var targetAttributesKnown : MutableMap<String, Int>) {
    var lastAttributeValue = 0

    fun insertTargetAttribute(targetAttribute : String) : Int? {
        if(!checkIfTargetAttributeIsKnown(targetAttribute)) {
            targetAttributesKnown.put(targetAttribute, lastAttributeValue)
            lastAttributeValue++
            return targetAttributesKnown.get(targetAttribute)
        }
        else {
            return targetAttributesKnown.get(targetAttribute)
        }
    }

    private fun checkIfTargetAttributeIsKnown(targetAttribute : String) : Boolean {
        return targetAttributesKnown.get(targetAttribute) != null
    }
}

fun main(args : Array<String>) {
}